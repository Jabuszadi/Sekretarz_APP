import React from 'react';
import PropTypes from 'prop-types';

function PageLayout({ title, description, actions, children, spacingClassName }) {
    return (
        <div className="px-6 py-8">
            <div className={`mx-auto flex w-full max-w-6xl flex-col ${spacingClassName}`}>
                {(title || description || actions) && (
                    <header className="flex flex-col gap-4 border-b border-gray-200 pb-5 md:flex-row md:items-center md:justify-between">
                        <div className="space-y-1">
                            {title && <h1 className="text-2xl font-semibold text-gray-900">{title}</h1>}
                            {description && <p className="text-sm text-gray-600">{description}</p>}
                        </div>
                        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
                    </header>
                )}
                <div className="flex flex-col gap-6">{children}</div>
            </div>
        </div>
    );
}

PageLayout.propTypes = {
    title: PropTypes.string,
    description: PropTypes.string,
    actions: PropTypes.node,
    children: PropTypes.node.isRequired,
    spacingClassName: PropTypes.string,
};

PageLayout.defaultProps = {
    title: undefined,
    description: undefined,
    actions: null,
    spacingClassName: 'gap-6',
};

export default PageLayout;

